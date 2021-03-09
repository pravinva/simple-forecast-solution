// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: MIT-0

import React, { useState, useEffect } from "react";
import styles from "./CustomerTable.module.scss";
import { CustomerInfo, AdminService } from "../../../services/admin";
import Table from "../../common/table/Table";
import { Panel } from "../../common/panel/Panel";
import { Preloader } from "../../common/preloader/Preloader";
import _ from "lodash";
import { TextField, Button } from "@material-ui/core";

const adminService = new AdminService();

export const CustomerTable: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [customers, setCustomers] = useState<CustomerInfo[]>();
  const [sorting, setSorting] = useState<keyof CustomerInfo>("registeredAt");
  const [direction, setDirection] = useState<"asc" | "desc">("desc");
  const [nextToken, setNextToken] = useState<string>();
  const [page, setPage] = useState<number>(0);
  const [rowsPerPage, setRowsPerPage] = useState<number>(10);
  const [searchFilter, setSearchFilter] = useState("");

  useEffect(() => {
    loadMoreCustomers();
  }, []);

  useEffect(() => {
    if (nextToken && page * rowsPerPage >= customers.length) {
      loadMoreCustomers();
    }
  }, [nextToken, page, rowsPerPage]);

  const loadMoreCustomers = async () => {
    if (loading) {
      return;
    }

    setLoading(true);

    try {
      const results = await adminService.getCustomerInfo({
        limit: 60,
        nextToken,
      });

      setCustomers((customers) => (customers || []).concat(results.customers));
      setNextToken(results.nextToken);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setCustomers(undefined);
    setNextToken(undefined);
    setPage(0);
  };

  const handleSorting = (newSorting: keyof CustomerInfo) => {
    setSorting(newSorting);
    setDirection(newSorting === sorting && direction === "asc" ? "desc" : "asc");
  };

  const handleSearchFilter = (evt) => {
    setSearchFilter(evt.target.value);
  };

  const handleSearch = () => {
    reset();
    loadMoreCustomers();
  };

  if (typeof customers === "undefined" || loading) {
    return (
      <div className={styles.container}>
        <Preloader />
      </div>
    );
  }

  let rows = _.sortBy(customers, (i) => i[sorting]);

  if (direction === "desc") {
    rows = rows.reverse();
  }

  rows = rows.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage);

  return (
    <div className={styles.container}>
      <Panel subTitle="Customers">
        {/* <div className={styles.search}>
          <TextField variant="outlined" label="Search" onChange={handleSearchFilter} value={searchFilter} />
          <Button variant="contained" color="secondary" onClick={handleSearch}>
            Search
          </Button>
        </div> */}
        <Table.Container>
          <Table.Head>
            <Table.Row>
              <Table.Cell>
                <Table.SortLabel
                  active={sorting === "name"}
                  direction={direction}
                  onClick={() => handleSorting("name")}
                >
                  Name
                </Table.SortLabel>
              </Table.Cell>
              <Table.Cell>
                <Table.SortLabel
                  active={sorting === "email"}
                  direction={direction}
                  onClick={() => handleSorting("email")}
                >
                  Email
                </Table.SortLabel>
              </Table.Cell>
              <Table.Cell>
                <Table.SortLabel
                  active={sorting === "noOfReports"}
                  direction={direction}
                  onClick={() => handleSorting("noOfReports")}
                >
                  Reports
                </Table.SortLabel>
              </Table.Cell>
              <Table.Cell>
                <Table.SortLabel
                  active={sorting === "lastReportAt"}
                  direction={direction}
                  onClick={() => handleSorting("lastReportAt")}
                >
                  Last Report At
                </Table.SortLabel>
              </Table.Cell>
              <Table.Cell>
                <Table.SortLabel
                  active={sorting === "registeredAt"}
                  direction={direction}
                  onClick={() => handleSorting("registeredAt")}
                >
                  Registered At
                </Table.SortLabel>
              </Table.Cell>
            </Table.Row>
          </Table.Head>
          <Table.Body>
            {rows.map((i) => (
              <Table.Row key={i.email}>
                <Table.Cell>{i.name}</Table.Cell>
                <Table.Cell>{i.email}</Table.Cell>
                <Table.Cell>{i.noOfReports}</Table.Cell>
                <Table.Cell>{i.lastReportAt?.toLocaleDateString() || "N/A"}</Table.Cell>
                <Table.Cell>{i.registeredAt?.toLocaleDateString() || "N/A"}</Table.Cell>
              </Table.Row>
            ))}
          </Table.Body>
        </Table.Container>
        <Table.Pagination
          rowsPerPageOptions={[10, 25, 50, 100, 200]}
          component="div"
          count={nextToken ? customers.length + rowsPerPage : customers.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onChangePage={(evt, i) => setPage(i)}
          onChangeRowsPerPage={(evt) => {
            setRowsPerPage(parseInt(evt.target.value, 10));
            setPage(0);
          }}
        />
      </Panel>
    </div>
  );
};
